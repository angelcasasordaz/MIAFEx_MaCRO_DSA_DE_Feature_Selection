"""Report dispatch remains explicit and bypasses MIAFEx scientific setup."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
import sys
import unittest
from unittest.mock import patch

import main_best as m


class DispatchTests(unittest.TestCase):
    def test_report_is_opt_in_for_plain_pycharm_run(self):
        self.assertFalse(m.REPORT_ONLY)
        self.assertFalse(m.parse_args([]).report_only)
        class StopAtNormalSetup(Exception):
            pass
        with patch.object(sys, 'argv', ['main_best.py']), \
                patch.object(m, 'resolve_execution_config', side_effect=StopAtNormalSetup) as backend, \
                patch('reporting.core.run_report') as report:
            with self.assertRaises(StopAtNormalSetup):
                m.main()
        backend.assert_called_once()
        report.assert_not_called()

    def test_report_aliases_bypass_every_scientific_path(self):
        for flag in ('--report-only', '--full-replica-report-only'):
            with self.subTest(flag=flag), ExitStack() as stack:
                stack.enter_context(patch.object(sys, 'argv', ['main_best.py', flag, '--exp-id', '604',
                                                               '--pipeline-mode', 'full', '--train-miafex', 'yes']))
                for name in ('resolve_execution_config', 'make_paths', 'run_single', 'execute_pending_runs',
                             'build_optimizer', 'train_miafex', 'extract_miafex_features', 'save_cache'):
                    mock = stack.enter_context(patch.object(m, name, side_effect=AssertionError('Forbidden science')))
                    self.addCleanup(mock.assert_not_called)
                report = stack.enter_context(patch('reporting.core.run_report', return_value={'ok': True}))
                self.assertEqual(m.main(), {'ok': True})
                report.assert_called_once()
                self.assertEqual(report.call_args.args[0].exp_id, 604)

    def test_help_exits_without_dispatch(self):
        with patch.object(sys, 'argv', ['main_best.py', '--help']), redirect_stdout(StringIO()), \
                patch('reporting.core.run_report') as report, patch.object(m, 'resolve_execution_config') as backend:
            with self.assertRaises(SystemExit) as exc:
                m.main()
        self.assertEqual(exc.exception.code, 0)
        report.assert_not_called()
        backend.assert_not_called()
