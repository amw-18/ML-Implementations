from utils import * 


lamdas = [0, np.exp(-18), 1]
degs = [2, 3, 6, 9]
# degs = [4]
size = [15, 300]


for s in size:
    regs = []
    for deg in degs:
        for lamda in lamdas:
            data_df = pd.read_csv("function1.csv",index_col = 0)
            test_size = int(s*2/3*0.25)
            x = data_df[:s].to_numpy()[:,:-1]
            y = data_df[:s].to_numpy()[:,-1]

            x_test, y_test = data_df[s:s+test_size].to_numpy()[:,:-1], data_df[s:s+test_size].to_numpy()[:,-1]

            reg, _, validation_error = crossVal(x, y, deg, "Polynomial", lamda=lamda)
            ytest_pred = reg.predict(x_test)
            rms_test = reg.rmsError(y_test, ytest_pred)
            regs.append((reg, validation_error, rms_test))
            # reg.Visualise()
    for reg in regs:
        # print(f"size: {s}, deg: {reg[0].deg}, lambda: {reg[0].lamda}, train error: {reg[0].rmsErr},val error: {reg[1]}, test error: {reg[2]}")
        print(f"{int(s*2/3)} & {np.around(reg[0].deg,3)} & {np.around(reg[0].lamda,3)} & {np.around(reg[0].rmsErr,3)} & {np.around(reg[1],3)} & {np.around(reg[2],3)}\\\\")

    best = min(regs, key=lambda x: x[1])
    # print(best[0].lamda, best[0].deg, best[1])
